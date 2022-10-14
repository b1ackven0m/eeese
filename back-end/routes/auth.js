const express = require("express")
const router = express.Router();
const mongoose = require("mongoose")
const Users = require("../models/users")
const bcrypt = require("bcrypt");
const URI = "mongodb+srv://depe:ufoarea51@cluster0.2vwc79u.mongodb.net/?retryWrites=true&w=majority"
mongoose.connect(URI,{useNewUrlParser:true})
.then(()=>{
    console.log("database connected")
}).catch((err)=> console.log(err))


router.post('/signup',async (req,res)=>{
    var user = req.body
    Users.findOne({phone : user.phone})
    .then((user)=>{
        res.json({msg:"user already exists"})
    }).catch((err)=>{
        console.log(err)
        res.status(500).json({msg:"server error"})
    })
    // user.password  = await bcrypt.hash(user.password,12)
    bcrypt.genSalt(10,(err, salt)=>{user.password,salt,()=>{
        if (err){
            res.json({msg:"there is an error hashing the password "})
        }
        user.password = hash
    }})
    console.log(user.password)
    newUser = new Users
    newUser.save()
    .then((data)=>{res.json({registered:data})})

    
})

module.exports = router